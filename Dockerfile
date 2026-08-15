FROM ruby:3.3

WORKDIR /site

# Install dependencies needed by Jekyll/GitHub Pages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY Gemfile Gemfile.lock* ./

# Install Ruby gems
RUN bundle install

# Copy site files
COPY . .

EXPOSE 4000

# Run Jekyll server with live reload
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload"]
